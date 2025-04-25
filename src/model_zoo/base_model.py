# hybrid-llm-inference/src/model_zoo/base_model.py
"""基础模型模块。"""

import torch
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from src.toolbox.logger import get_logger

logger = get_logger(__name__)

class BaseModel(ABC):
    """基础模型类，提供通用的模型接口。"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化模型。
        
        Args:
            config: 模型配置
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"模型初始化完成，使用设备: {self.device}")
    
    def load_model(self):
        """加载模型。"""
        raise NotImplementedError
    
    def get_token_count(self, text: str) -> int:
        """获取文本的token数量。
        
        Args:
            text: 输入文本
            
        Returns:
            token数量
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer未初始化")
        return len(self.tokenizer.encode(text))
    
    def infer(self, prompt: str, max_tokens: int = 100) -> str:
        """执行推理。
        
        Args:
            prompt: 输入提示
            max_tokens: 最大生成token数
            
        Returns:
            生成的文本
        """
        if self.model is None:
            raise RuntimeError("模型未加载")
        
        start_time = time.time()
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"推理完成，耗时: {time.time() - start_time:.2f}秒")
            return response
        except Exception as e:
            logger.error(f"推理失败: {str(e)}")
            raise
    
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
        
    def initialize(self):
        """初始化模型。"""
        if self.model is None:
            self.model = self.load_model()
            self.tokenizer = self.tokenizer
            logger.info("模型初始化完成")
        else:
            logger.info("模型已初始化")
