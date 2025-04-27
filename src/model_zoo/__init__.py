# hybrid-llm-inference/src/model_zoo/__init__.py
"""模型库模块。"""

from typing import Dict, Type
import logging
from .base_model import BaseModel
from .mock_model import MockModel

logger = logging.getLogger(__name__)

# 注册所有可用的模型
MODELS: Dict[str, Type[BaseModel]] = {
    "mock": MockModel,
}

def get_model(model_name: str, model_path: str) -> BaseModel:
    """获取模型实例。
    
    Args:
        model_name: 模型名称
        model_path: 模型路径
        
    Returns:
        BaseModel: 模型实例
    """
    if model_name not in MODELS:
        raise ValueError(f"不支持的模型: {model_name}")
        
    try:
        return MODELS[model_name](model_path)
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        raise

__all__ = ["BaseModel", "MockModel", "get_model"]
