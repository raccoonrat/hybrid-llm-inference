import os
import pytest
import yaml
from pathlib import Path
from src.toolbox.config_manager import ConfigManager

@pytest.fixture
def config_dir(tmp_path):
    """创建临时配置文件目录"""
    # 创建硬件配置文件
    hardware_config = {
        "devices": {
            "rtx4050": {
                "device_type": "gpu",
                "device_id": 0,
                "idle_power": 15.0,
                "memory_limit": 6144,
                "compute_capability": 8.9,
                "priority": 1
            }
        }
    }
    
    # 创建模型配置文件
    model_config = {
        "models": {
            "llama3": {
                "model_name": "meta-llama/Llama-3-8B",
                "mode": "local",
                "max_length": 512,
                "mixed_precision": "fp16",
                "device_placement": True
            }
        }
    }
    
    # 创建调度器配置文件
    scheduler_config = {
        "scheduler": {
            "max_batch_size": 4,
            "max_queue_size": 100,
            "max_wait_time": 1.0,
            "token_threshold": 512,
            "dynamic_threshold": True,
            "batch_processing": True,
            "device_priority": ["rtx4050"],
            "monitoring": {
                "sample_interval": 200,
                "metrics": ["power_usage", "memory_usage"]
            }
        }
    }
    
    # 写入配置文件
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    
    with open(config_dir / "hardware_config.yaml", "w") as f:
        yaml.dump(hardware_config, f)
    
    with open(config_dir / "model_config.yaml", "w") as f:
        yaml.dump(model_config, f)
    
    with open(config_dir / "scheduler_config.yaml", "w") as f:
        yaml.dump(scheduler_config, f)
    
    return str(config_dir)

def test_load_valid_configs(config_dir):
    """测试正常加载配置文件"""
    config_manager = ConfigManager(config_dir)
    
    # 加载硬件配置
    hardware_config = config_manager.load_config("hardware_config.yaml")
    assert "devices" in hardware_config
    assert "rtx4050" in hardware_config["devices"]
    
    # 加载模型配置
    model_config = config_manager.load_config("model_config.yaml")
    assert "models" in model_config
    assert "llama3" in model_config["models"]
    
    # 加载调度器配置
    scheduler_config = config_manager.load_config("scheduler_config.yaml")
    assert "scheduler" in scheduler_config
    assert "device_priority" in scheduler_config["scheduler"]

def test_load_nonexistent_config(config_dir):
    """测试加载不存在的配置文件"""
    config_manager = ConfigManager(config_dir)
    
    with pytest.raises(FileNotFoundError):
        config_manager.load_config("nonexistent_config.yaml")

def test_load_invalid_yaml(config_dir):
    """测试加载格式错误的YAML文件"""
    # 创建一个格式错误的YAML文件
    invalid_config_path = Path(config_dir) / "invalid_config.yaml"
    with open(invalid_config_path, "w") as f:
        f.write("invalid: yaml: content")
    
    config_manager = ConfigManager(config_dir)
    
    with pytest.raises(ValueError, match="配置文件格式错误"):
        config_manager.load_config("invalid_config.yaml")

def test_validate_invalid_hardware_config(config_dir):
    """测试验证无效的硬件配置"""
    # 创建一个缺少必要字段的硬件配置
    invalid_hardware_config = {
        "devices": {
            "rtx4050": {
                "device_type": "gpu",
                # 缺少其他必要字段
            }
        }
    }
    
    config_path = Path(config_dir) / "invalid_hardware_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(invalid_hardware_config, f)
    
    config_manager = ConfigManager(config_dir)
    
    with pytest.raises(ValueError, match="设备配置缺少必要字段"):
        config_manager.load_config("invalid_hardware_config.yaml")

def test_validate_invalid_model_config(config_dir):
    """测试验证无效的模型配置"""
    # 创建一个缺少必要字段的模型配置
    invalid_model_config = {
        "models": {
            "llama3": {
                "model_name": "meta-llama/Llama-3-8B",
                # 缺少其他必要字段
            }
        }
    }
    
    config_path = Path(config_dir) / "invalid_model_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(invalid_model_config, f)
    
    config_manager = ConfigManager(config_dir)
    
    with pytest.raises(ValueError, match="模型配置缺少必要字段"):
        config_manager.load_config("invalid_model_config.yaml")

def test_validate_invalid_scheduler_config(config_dir):
    """测试验证无效的调度器配置"""
    # 创建一个缺少必要字段的调度器配置
    invalid_scheduler_config = {
        "scheduler": {
            "max_batch_size": 4,
            # 缺少其他必要字段
        }
    }
    
    config_path = Path(config_dir) / "invalid_scheduler_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(invalid_scheduler_config, f)
    
    config_manager = ConfigManager(config_dir)
    
    with pytest.raises(ValueError, match="调度器配置缺少必要字段"):
        config_manager.load_config("invalid_scheduler_config.yaml")

def test_get_unloaded_config(config_dir):
    """测试获取未加载的配置"""
    config_manager = ConfigManager(config_dir)
    
    with pytest.raises(ValueError, match="配置未加载"):
        config_manager.get_config("hardware_config.yaml")
    
    # 加载配置后再获取
    config_manager.load_config("hardware_config.yaml")
    hardware_config = config_manager.get_config("hardware_config.yaml")
    assert "devices" in hardware_config 