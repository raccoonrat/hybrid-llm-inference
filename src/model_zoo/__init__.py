# hybrid-llm-inference/src/model_zoo/__init__.py
"""模型库模块。"""

from typing import Dict, Type, Union, Any
import logging
from .base_model import BaseModel
from .mock_model import MockModel
from .tinyllama import TinyLlama

logger = logging.getLogger(__name__)

# 注册所有可用的模型
MODELS: Dict[str, Type[BaseModel]] = {
    "mock": MockModel,
    "TinyLlama-1.1B-Chat-v1.0": TinyLlama
}

def get_model(model_name: str, model_path: Union[str, Dict[str, Any]]) -> BaseModel:
    """获取模型实例。
    
    Args:
        model_name: 模型名称
        model_path: 模型路径或配置字典
        
    Returns:
        BaseModel: 模型实例
    """
    if model_name not in MODELS:
        raise ValueError(f"不支持的模型: {model_name}")
        
    try:
        # 如果 model_path 是字符串，将其转换为配置字典
        config = {"model_path": model_path} if isinstance(model_path, str) else model_path
        return MODELS[model_name](config)
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        raise

__all__ = ["BaseModel", "MockModel", "get_model"]
