# hybrid-llm-inference/tests/unit/test_scheduling.py
import pytest
import os
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
    """硬件配置"""
    return {
        "m1_pro": {"type": "cpu_gpu", "idle_power": 10.0},
        "a100": {"type": "gpu", "device_id": 0},
        "rtx4050": {
            "type": "gpu",
            "device_id": 0,
            "idle_power": 15.0,
            "sample_interval": 200
        }
    }

@pytest.fixture
def model_config():
    """模型配置"""
    return {
        "model_name": "tinyllama",
        "model_path": "models/TinyLlama-1.1B-Chat-v1.0",
        "mode": "local",
        "batch_size": 1,
        "max_length": 128
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

def test_task_allocator(hardware_config, model_config):
    """测试任务分配器"""
    # 设置测试模式环境变量
    os.environ['TEST_MODE'] = '1'
    
    allocator = TaskAllocator(
        hardware_config=hardware_config,
        model_config=model_config
    )
    
    # 测试任务分配
    task = {
        "input_tokens": 32,
        "output_tokens": 32
    }
    
    device = allocator.allocate(task)
    assert device in ["m1_pro", "a100", "rtx4050"]

def test_task_allocator_invalid_hardware(hardware_config, model_config):
    """测试无效硬件配置"""
    # 设置测试模式环境变量
    os.environ['TEST_MODE'] = '1'
    
    invalid_config = hardware_config.copy()
    invalid_config["invalid"] = {"type": "invalid"}
    
    with pytest.raises(ValueError, match="Invalid hardware type"):
        TaskAllocator(
            hardware_config=invalid_config,
            model_config=model_config
        )
