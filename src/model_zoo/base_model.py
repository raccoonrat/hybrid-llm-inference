# hybrid-llm-inference/src/model_zoo/base_model.py
"""基础模型模块。"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union
from toolbox.logger import get_logger
import logging
from pathlib import Path

logger = get_logger(__name__)

class BaseModel(ABC):
    """基础模型类，提供通用的模型接口。"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化基础模型。
        
        Args:
            config: 配置字典，包含：
                - model_path: 模型路径
                - device: 设备类型，可选 "cuda" 或 "cpu"
                - 其他特定模型的配置
        """
        self.logger = logging.getLogger(__name__)
        
        if isinstance(config, str):
            self.config = {"model_path": config}
        else:
            self.config = config
            
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = self.config.get("model_path", "")
        
        # 检查是否在测试模式下
        if os.environ.get("TEST_MODE", "").lower() == "true":
            self.logger.info("测试模式：跳过模型加载")
            return
            
        if not self.model_path:
            raise ValueError("model_path 不能为空")
            
        self._load_model()
        
    def _load_model(self) -> None:
        """加载模型。子类需要实现此方法。"""
        raise NotImplementedError("子类必须实现 _load_model 方法")
    
    def get_token_count(self, text: str) -> int:
        """获取文本的token数量。
        
        Args:
            text: 输入文本
            
        Returns:
            int: token数量
        """
        if self.model_path:
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                tokens = tokenizer.encode(text)
                return len(tokens)
            except Exception as e:
                self.logger.error(f"计算token数量失败: {str(e)}")
                return 0
        else:
            return len(text.split())
    
    def infer(self, input_text: str, **kwargs) -> str:
        """执行推理。子类需要实现此方法。

        Args:
            input_text: 输入文本
            **kwargs: 其他推理参数
        Returns:
            str: 输出文本
        """
        raise NotImplementedError("子类必须实现 infer 方法，且需支持 **kwargs")
    
    @abstractmethod
    def _validate_base_config(self) -> None:
        """验证基础配置。"""
        pass
    
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
    
    def cleanup(self) -> None:
        """清理资源。子类可以重写此方法。"""
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
        raise NotImplementedError
        
    def get_metrics(self) -> Dict[str, float]:
        """获取性能指标。
        
        Returns:
            Dict[str, float]: 包含以下指标:
                - total_tokens: 处理的总token数
                - total_time: 总运行时间（秒）
                - avg_tokens_per_second: 平均每秒处理的token数
                - avg_time_per_call: 平均每次调用时间（秒）
        """
        return {
            "total_tokens": 0,
            "total_time": 0.0,
            "avg_tokens_per_second": 0.0,
            "avg_time_per_call": 0.0
        }
        
    def reset_metrics(self) -> None:
        """重置性能指标。"""
        self.total_tokens = 0
        self.total_time = 0.0
        self.call_count = 0
        
    def initialize(self):
        """初始化模型。"""
        if self.model_path:
            self._load_model()
            logger.info("模型初始化完成")
        else:
            logger.info("模型已初始化")

    @abstractmethod
    def generate(self, input_text: str, max_tokens: Optional[int] = None, temperature: float = 0.7) -> str:
        """生成文本。

        Args:
            input_text: 输入文本
            max_tokens: 最大生成令牌数
            temperature: 采样温度，控制生成的随机性

        Returns:
            str: 生成的文本
        """
        pass
