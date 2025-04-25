# hybrid-llm-inference/src/model_zoo/__init__.py
"""模型库模块。"""

import logging
from typing import Dict, Any, Optional

from .base_model import BaseModel
from .tinyllama import TinyLlama
from .falcon import FalconModel
from .mistral import LocalMistral, APIMistral
from .llama3 import LocalLlama3, APILlama3

logger = logging.getLogger(__name__)

def get_model(config: Dict[str, Any]) -> BaseModel:
    """获取模型实例。
    
    Args:
        config: 配置字典，必须包含：
            - model_name: 模型名称
            - model_path: 模型路径（本地模式）或模型 ID（远程模式）
            - mode: 运行模式（"local" 或 "remote"）
            - batch_size: 批处理大小
            - max_length: 最大长度
            - device: 运行设备（可选，默认为 "auto"）
            - dtype: 数据类型（可选，默认为 "auto"）
            
    Returns:
        BaseModel: 模型实例
        
    Raises:
        ValueError: 当配置无效时
    """
    # 验证配置
    required_fields = ["model_name", "model_path", "mode", "batch_size", "max_length"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"缺少必需的配置字段: {field}")
    
    # 根据模型名称和模式选择模型类
    model_name = config["model_name"].lower()
    mode = config["mode"].lower()
    
    if model_name == "tinyllama":
        return TinyLlama(config)
            
    elif model_name == "falcon":
        if mode == "local":
            return FalconModel(config)
        else:
            raise ValueError("Falcon 只支持本地模式")
            
    elif model_name == "mistral":
        if mode == "local":
            return LocalMistral(config)
        elif mode == "remote":
            return APIMistral(config)
        else:
            raise ValueError("Mistral 只支持本地和远程模式")
            
    elif model_name == "llama3":
        if mode == "local":
            return LocalLlama3(config)
        elif mode == "remote":
            return APILlama3(config)
        else:
            raise ValueError("Llama3 只支持本地和远程模式")
            
    else:
        raise ValueError(f"不支持的模型: {model_name}")

__all__ = ["BaseModel", "TinyLlama", "get_model"]
