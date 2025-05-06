import os
import pytest
import yaml
from pathlib import Path
from src.toolbox.config_manager import ConfigManager

@pytest.fixture
def config_dir(tmp_path):
    """创建临时配置文件目录，使用标准格式配置"""
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
                "model_path": "/path/to/model",
                "device": "cuda",
                "dtype": "float32",
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

@pytest.fixture
def pipeline_style_config():
    """创建 SystemPipeline 风格的配置字典"""
    return {
        "model": {
            "model_name": "TinyLlama-1.1B-Chat-v1.0",
            "model_path": "models/TinyLlama-1.1B-Chat-v1.0",
            "device": "cuda",
            "dtype": "float32",
            "batch_size": 1,
            "max_length": 512
        },
        "hardware": {
            "device_type": "gpu",
            "device_id": 0,
            "idle_power": 15.0,
            "sample_interval": 200,
            "device": "cuda"
        },
        "scheduler": {
            "scheduler_type": "token_based",
            "max_batch_size": 4,
            "max_queue_size": 100,
            "max_wait_time": 1.0
        }
    }

@pytest.fixture
def standard_style_config():
    """创建标准格式的配置字典"""
    return {
        "hardware_config": {
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
        },
        "model_config": {
            "models": {
                "llama3": {
                    "model_name": "meta-llama/Llama-3-8B",
                    "model_path": "/path/to/model",
                    "device": "cuda",
                    "dtype": "float32",
                    "mode": "local",
                    "max_length": 512,
                    "mixed_precision": "fp16",
                    "device_placement": True
                }
            }
        },
        "scheduler_config": {
            "scheduler": {
                "max_batch_size": 4,
                "max_queue_size": 100,
                "max_wait_time": 1.0,
                "token_threshold": 512,
                "dynamic_threshold": True,
                "batch_processing": True,
                "device_priority": ["rtx4050"]
            }
        }
    }

def test_config_manager_with_directory(config_dir):
    """测试使用配置目录初始化配置管理器"""
    config_manager = ConfigManager(config_dir)
    
    # 测试加载硬件配置
    hardware_config = config_manager.load_config("hardware_config.yaml")
    assert "devices" in hardware_config
    assert "rtx4050" in hardware_config["devices"]
    
    # 测试加载模型配置
    model_config = config_manager.load_config("model_config.yaml")
    assert "models" in model_config
    assert "llama3" in model_config["models"]
    
    # 测试加载调度器配置
    scheduler_config = config_manager.load_config("scheduler_config.yaml")
    assert "scheduler" in scheduler_config
    assert scheduler_config["scheduler"]["max_batch_size"] == 4

def test_config_manager_with_pipeline_style_dict(pipeline_style_config):
    """测试使用 SystemPipeline 风格的配置字典初始化配置管理器"""
    config_manager = ConfigManager(pipeline_style_config)
    
    # 测试转换后的硬件配置
    hardware_config = config_manager.get_hardware_config()
    assert "devices" in hardware_config
    assert "rtx4050" in hardware_config["devices"]
    assert hardware_config["devices"]["rtx4050"]["device_type"] == "gpu"
    assert hardware_config["devices"]["rtx4050"]["device_id"] == 0
    assert hardware_config["devices"]["rtx4050"]["idle_power"] == 15.0
    assert hardware_config["devices"]["rtx4050"]["memory_limit"] == 6144
    assert hardware_config["devices"]["rtx4050"]["compute_capability"] == 8.9
    assert hardware_config["devices"]["rtx4050"]["priority"] == 1
    assert hardware_config["devices"]["rtx4050"]["sample_interval"] == 200
    
    # 测试转换后的模型配置
    model_config = config_manager.get_model_config()
    assert "models" in model_config
    assert "TinyLlama-1.1B-Chat-v1.0" in model_config["models"]
    model = model_config["models"]["TinyLlama-1.1B-Chat-v1.0"]
    assert model["model_name"] == "TinyLlama-1.1B-Chat-v1.0"
    assert model["model_path"] == "models/TinyLlama-1.1B-Chat-v1.0"
    assert model["device"] == "cuda"
    assert model["dtype"] == "float32"
    assert model["mode"] == "local"
    assert model["max_length"] == 512
    assert model["mixed_precision"] == "fp16"
    assert model["device_placement"] is True
    assert model["batch_size"] == 1
    
    # 测试转换后的调度器配置
    scheduler_config = config_manager.get_scheduler_config()
    assert "scheduler" in scheduler_config
    scheduler = scheduler_config["scheduler"]
    assert scheduler["max_batch_size"] == 4
    assert scheduler["max_queue_size"] == 100
    assert scheduler["max_wait_time"] == 1.0
    assert scheduler["token_threshold"] == 512
    assert scheduler["dynamic_threshold"] is True
    assert scheduler["batch_processing"] is True
    assert scheduler["device_priority"] == ["rtx4050"]
    assert "monitoring" in scheduler
    assert scheduler["monitoring"]["sample_interval"] == 200
    assert scheduler["monitoring"]["metrics"] == ["power_usage", "memory_usage"]

def test_config_manager_with_standard_style_dict(standard_style_config):
    """测试使用标准格式的配置字典初始化配置管理器"""
    config_manager = ConfigManager(standard_style_config)
    
    # 测试硬件配置
    hardware_config = config_manager.get_hardware_config()
    assert "devices" in hardware_config
    assert "rtx4050" in hardware_config["devices"]
    assert hardware_config["devices"]["rtx4050"]["device_type"] == "gpu"
    
    # 测试模型配置
    model_config = config_manager.get_model_config()
    assert "models" in model_config
    assert "llama3" in model_config["models"]
    assert model_config["models"]["llama3"]["mode"] == "local"
    
    # 测试调度器配置
    scheduler_config = config_manager.get_scheduler_config()
    assert "scheduler" in scheduler_config
    assert scheduler_config["scheduler"]["max_batch_size"] == 4

def test_invalid_hardware_config():
    """测试无效的硬件配置"""
    invalid_config = {
        "hardware_config": {
            "devices": {
                "rtx4050": {
                    "device_type": 123,  # 应该是字符串
                    "device_id": "0",    # 应该是整数
                    "idle_power": "15.0" # 应该是数值
                }
            }
        }
    }
    
    with pytest.raises(ValueError):
        ConfigManager(invalid_config)

def test_invalid_model_config():
    """测试无效的模型配置"""
    invalid_config = {
        "model_config": {
            "models": {
                "llama3": {
                    "model_name": 123,    # 应该是字符串
                    "mode": ["local"],    # 应该是字符串
                    "max_length": "512"   # 应该是整数
                }
            }
        }
    }
    
    with pytest.raises(ValueError):
        ConfigManager(invalid_config)

def test_invalid_scheduler_config():
    """测试无效的调度器配置"""
    invalid_config = {
        "scheduler_config": {
            "scheduler": {
                "max_batch_size": "4",    # 应该是整数
                "max_queue_size": "100",  # 应该是整数
                "max_wait_time": "1.0"    # 应该是数值
            }
        }
    }
    
    with pytest.raises(ValueError):
        ConfigManager(invalid_config)

def test_missing_required_fields():
    """测试缺少必需字段"""
    invalid_config = {
        "hardware_config": {
            "devices": {
                "rtx4050": {
                    "device_type": "gpu"
                    # 缺少 device_id 和 idle_power
                }
            }
        }
    }
    
    with pytest.raises(ValueError):
        ConfigManager(invalid_config)

def test_config_file_not_found(config_dir):
    """测试配置文件不存在的情况"""
    config_manager = ConfigManager(config_dir)
    
    with pytest.raises(FileNotFoundError):
        config_manager.load_config("nonexistent.yaml")

def test_pipeline_style_config_with_minimal_values():
    """测试使用最小配置的 SystemPipeline 风格配置"""
    minimal_config = {
        "model": {
            "model_name": "test-model",
            "model_path": "/path/to/model"
        },
        "hardware": {
            "device_type": "gpu",
            "device_id": 0,
            "idle_power": 10.0
        },
        "scheduler": {
            "scheduler_type": "token_based"
        }
    }
    
    config_manager = ConfigManager(minimal_config)
    
    # 验证默认值是否正确设置
    hardware_config = config_manager.get_hardware_config()
    device = hardware_config["devices"]["rtx4050"]
    assert device["memory_limit"] == 6144
    assert device["compute_capability"] == 8.9
    assert device["priority"] == 1
    assert device["sample_interval"] == 200
    
    model_config = config_manager.get_model_config()
    model = model_config["models"]["test-model"]
    assert model["device"] == "cuda"
    assert model["dtype"] == "float32"
    assert model["mode"] == "local"
    assert model["max_length"] == 512
    assert model["mixed_precision"] == "fp16"
    assert model["device_placement"] is True
    assert model["batch_size"] == 1
    
    scheduler_config = config_manager.get_scheduler_config()
    scheduler = scheduler_config["scheduler"]
    assert scheduler["max_batch_size"] == 4
    assert scheduler["max_queue_size"] == 100
    assert scheduler["max_wait_time"] == 1.0
    assert scheduler["token_threshold"] == 512
    assert scheduler["dynamic_threshold"] is True
    assert scheduler["batch_processing"] is True

def test_pipeline_style_config_with_custom_values():
    """测试使用自定义值的 SystemPipeline 风格配置"""
    custom_config = {
        "model": {
            "model_name": "custom-model",
            "model_path": "/custom/path",
            "device": "cpu",
            "dtype": "float16",
            "mode": "remote",
            "max_length": 1024,
            "mixed_precision": "fp32",
            "device_placement": False,
            "batch_size": 2
        },
        "hardware": {
            "device_type": "cpu",
            "device_id": 1,
            "idle_power": 5.0,
            "memory_limit": 8192,
            "compute_capability": 7.5,
            "priority": 2,
            "sample_interval": 500
        },
        "scheduler": {
            "scheduler_type": "token_based",
            "max_batch_size": 8,
            "max_queue_size": 200,
            "max_wait_time": 2.0,
            "token_threshold": 1024,
            "dynamic_threshold": False,
            "batch_processing": False
        }
    }
    
    config_manager = ConfigManager(custom_config)
    
    # 验证自定义值是否正确保留
    hardware_config = config_manager.get_hardware_config()
    device = hardware_config["devices"]["rtx4050"]
    assert device["device_type"] == "cpu"
    assert device["device_id"] == 1
    assert device["idle_power"] == 5.0
    assert device["memory_limit"] == 8192
    assert device["compute_capability"] == 7.5
    assert device["priority"] == 2
    assert device["sample_interval"] == 500
    
    model_config = config_manager.get_model_config()
    model = model_config["models"]["custom-model"]
    assert model["device"] == "cpu"
    assert model["dtype"] == "float16"
    assert model["mode"] == "remote"
    assert model["max_length"] == 1024
    assert model["mixed_precision"] == "fp32"
    assert model["device_placement"] is False
    assert model["batch_size"] == 2
    
    scheduler_config = config_manager.get_scheduler_config()
    scheduler = scheduler_config["scheduler"]
    assert scheduler["max_batch_size"] == 8
    assert scheduler["max_queue_size"] == 200
    assert scheduler["max_wait_time"] == 2.0
    assert scheduler["token_threshold"] == 1024
    assert scheduler["dynamic_threshold"] is False
    assert scheduler["batch_processing"] is False 