"""模型库模块。"""

import os
from typing import Dict, Any, Optional
from .base_model import BaseModel
from toolbox.logger import get_logger

logger = get_logger(__name__)

def get_model(model_name: str, mode: str, config: Dict[str, Any]) -> BaseModel:
    """获取模型实例。

    Args:
        model_name: 模型名称
        mode: 运行模式
        config: 模型配置

    Returns:
        BaseModel: 模型实例
    """
    # 在测试模式下返回模拟模型
    if os.getenv("TEST_MODE") == "true":
        from .mock_model import MockModel
        return MockModel(config)
    
    # 根据模型名称返回对应的模型实例
    if model_name == "tinyllama":
        from .tinyllama import TinyLlama
        return TinyLlama(config)
    elif model_name == "mistral":
        from .mistral import Mistral
        return Mistral(config)
    else:
        raise ValueError(f"不支持的模型: {model_name}") 