# hybrid-llm-inference/src/model_zoo/base_model.py
"""基础模型模块。"""

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from toolbox.logger import get_logger

logger = get_logger(__name__)

class BaseModel(ABC):
    """基础模型类。"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化基础模型。

        Args:
            config: 配置字典
        """
        self.config = config
        self.initialized = False
        
        # 验证基础配置
        self._validate_base_config()
        
        logger.info("基础模型初始化完成")
    
    def _validate_base_config(self) -> None:
        """验证基础配置。"""
        if not isinstance(self.config, dict):
            raise ValueError("配置必须是字典类型")
    
    @abstractmethod
    def _validate_config(self) -> None:
        """验证配置。

        由子类实现，用于验证特定模型的配置。
        """
        pass
    
    @abstractmethod
    def _init_model(self) -> None:
        """初始化模型。

        由子类实现，用于初始化特定模型。
        """
        pass
    
    @abstractmethod
    def inference(self, input_text: str, max_tokens: Optional[int] = None) -> str:
        """执行推理。

        Args:
            input_text: 输入文本
            max_tokens: 最大生成令牌数

        Returns:
            生成的文本
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """清理资源。

        由子类实现，用于清理特定模型的资源。
        """
        pass

    def _do_inference(self, input_text: str) -> str:
        """执行实际的推理操作。
        
        Args:
            input_text: 输入文本
            
        Returns:
            str: 生成的文本
            
        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        pass
        
    def get_metrics(self) -> Dict[str, float]:
        """获取性能指标。
        
        Returns:
            Dict[str, float]: 包含以下指标:
                - total_tokens: 处理的总token数
                - total_time: 总运行时间（秒）
                - avg_tokens_per_second: 平均每秒处理的token数
                - avg_time_per_call: 平均每次调用时间（秒）
        """
        pass
        
    def reset_metrics(self) -> None:
        """重置性能指标。"""
        pass
        
    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """获取文本的token数量。
        
        Args:
            text: 输入文本
            
        Returns:
            int: token数量
            
        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        pass
        
    def initialize(self):
        """初始化模型。"""
        if self.initialized:
            return
        self.initialized = True
        logger.info("模型初始化完成")
        
    def infer(self, text: str) -> str:
        """
        执行推理的包装方法。

        Args:
            text: 输入文本

        Returns:
            str: 输出文本
        """
        return self.inference(text)
