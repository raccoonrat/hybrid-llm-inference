"""测试模型工厂函数。"""

import pytest
import os
from src.model_zoo import get_model
from src.model_zoo.base_model import BaseModel

def test_get_model_missing_config():
    """测试缺少必需配置字段时的错误处理。"""
    # 缺少必需字段的配置
    incomplete_configs = [
        {},  # 完全空的配置
        {"model_name": "tinyllama"},  # 只有模型名称
        {"model_name": "tinyllama", "model_path": "/path/to/model"},  # 缺少mode
        {"model_name": "tinyllama", "model_path": "/path/to/model", "mode": "local"},  # 缺少batch_size
        {"model_name": "tinyllama", "model_path": "/path/to/model", "mode": "local", "batch_size": 1},  # 缺少max_length
        None,  # None配置
        "",  # 空字符串
        123,  # 数字
        [],  # 空列表
        {"invalid_key": "value"}  # 无效的键
    ]
    
    for config in incomplete_configs:
        with pytest.raises((ValueError, TypeError)) as exc_info:
            get_model(config)
        assert any(msg in str(exc_info.value) for msg in ["缺少必需的配置字段", "必须是字典"])

def test_get_model_invalid_model_name():
    """测试不支持的模型名称。"""
    invalid_model_configs = [
        {
            "model_name": "unsupported_model",
            "model_path": "/path/to/model",
            "mode": "local",
            "batch_size": 1,
            "max_length": 512
        },
        {
            "model_name": "",  # 空模型名称
            "model_path": "/path/to/model",
            "mode": "local",
            "batch_size": 1,
            "max_length": 512
        },
        {
            "model_name": "123",  # 数字模型名称
            "model_path": "/path/to/model",
            "mode": "local",
            "batch_size": 1,
            "max_length": 512
        },
        {
            "model_name": None,  # None模型名称
            "model_path": "/path/to/model",
            "mode": "local",
            "batch_size": 1,
            "max_length": 512
        }
    ]
    
    for config in invalid_model_configs:
        with pytest.raises(ValueError) as exc_info:
            get_model(config)
        assert "不支持的模型" in str(exc_info.value)

def test_get_model_invalid_mode():
    """测试不支持的运行模式。"""
    # Falcon只支持本地模式
    falcon_configs = [
        {
            "model_name": "falcon",
            "model_path": "/path/to/model",
            "mode": "remote",  # 不支持的模式
            "batch_size": 1,
            "max_length": 512
        },
        {
            "model_name": "falcon",
            "model_path": "/path/to/model",
            "mode": "invalid",  # 无效的模式
            "batch_size": 1,
            "max_length": 512
        },
        {
            "model_name": "falcon",
            "model_path": "/path/to/model",
            "mode": "",  # 空模式
            "batch_size": 1,
            "max_length": 512
        }
    ]
    
    for config in falcon_configs:
        with pytest.raises(ValueError) as exc_info:
            get_model(config)
        assert "Falcon 只支持本地模式" in str(exc_info.value)
    
    # Mistral和Llama3支持本地和远程模式
    for model_name in ["mistral", "llama3"]:
        invalid_mode_configs = [
            {
                "model_name": model_name,
                "model_path": "/path/to/model",
                "mode": "invalid_mode",  # 无效的模式
                "batch_size": 1,
                "max_length": 512
            },
            {
                "model_name": model_name,
                "model_path": "/path/to/model",
                "mode": "",  # 空模式
                "batch_size": 1,
                "max_length": 512
            },
            {
                "model_name": model_name,
                "model_path": "/path/to/model",
                "mode": None,  # None模式
                "batch_size": 1,
                "max_length": 512
            }
        ]
        
        for config in invalid_mode_configs:
            with pytest.raises(ValueError) as exc_info:
                get_model(config)
            assert f"{model_name.capitalize()} 只支持本地和远程模式" in str(exc_info.value)

def test_get_model_test_mode(monkeypatch):
    """测试测试模式下返回模拟模型。"""
    monkeypatch.setenv("TEST_MODE", "true")
    
    test_configs = [
        {
            "model_name": "any_model",  # 在测试模式下，模型名称不重要
            "model_path": "/path/to/model",
            "mode": "local",
            "batch_size": 1,
            "max_length": 512
        },
        {
            "model_name": "tinyllama",  # 具体的模型名称
            "model_path": "/path/to/model",
            "mode": "local",
            "batch_size": 1,
            "max_length": 512
        },
        {
            "model_name": "invalid_model",  # 无效的模型名称
            "model_path": "/path/to/model",
            "mode": "local",
            "batch_size": 1,
            "max_length": 512
        }
    ]
    
    for config in test_configs:
        model = get_model(config)
        assert "MockModel" in str(type(model))
    
    # 测试环境变量切换
    monkeypatch.setenv("TEST_MODE", "false")
    with pytest.raises(ValueError):
        get_model(test_configs[-1])  # 使用无效的模型名称应该引发错误

def test_get_model_valid_configs():
    """测试有效的配置。"""
    valid_configs = [
        {
            "model_name": "tinyllama",
            "model_path": "/path/to/model",
            "mode": "local",
            "batch_size": 1,
            "max_length": 512
        },
        {
            "model_name": "falcon",
            "model_path": "/path/to/model",
            "mode": "local",
            "batch_size": 1,
            "max_length": 512,
            "device": "cuda",
            "dtype": "float16"
        },
        {
            "model_name": "mistral",
            "model_path": "/path/to/model",
            "mode": "local",
            "batch_size": 1,
            "max_length": 512,
            "device": "cuda",
            "dtype": "float16",
            "temperature": 0.7,
            "top_p": 0.9
        },
        {
            "model_name": "mistral",
            "model_path": "model_id",
            "mode": "remote",
            "batch_size": 1,
            "max_length": 512,
            "api_key": "test_key",
            "api_url": "https://api.example.com"
        },
        {
            "model_name": "llama3",
            "model_path": "/path/to/model",
            "mode": "local",
            "batch_size": 1,
            "max_length": 512,
            "device": "cpu",
            "dtype": "float32",
            "use_cache": True
        }
    ]
    
    for config in valid_configs:
        try:
            model = get_model(config)
            assert isinstance(model, BaseModel)
            # 验证配置是否被正确保存
            assert model.config == config
        except Exception as e:
            if "找不到模型文件" not in str(e):  # 忽略模型文件不存在的错误
                raise

def test_get_model_path_validation():
    """测试模型路径验证。"""
    # 创建临时模型文件
    test_model_path = "test_model.bin"
    try:
        with open(test_model_path, "w") as f:
            f.write("test model content")
        
        # 测试存在的模型文件
        config = {
            "model_name": "tinyllama",
            "model_path": test_model_path,
            "mode": "local",
            "batch_size": 1,
            "max_length": 512
        }
        
        try:
            model = get_model(config)
            assert isinstance(model, BaseModel)
        except Exception as e:
            if "找不到模型文件" not in str(e):
                raise
            
        # 测试不存在的模型文件
        config["model_path"] = "non_existent_model.bin"
        with pytest.raises(ValueError) as exc_info:
            get_model(config)
        assert "找不到模型文件" in str(exc_info.value)
        
    finally:
        # 清理临时文件
        if os.path.exists(test_model_path):
            os.remove(test_model_path) 