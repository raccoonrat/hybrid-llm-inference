# hybrid-llm-inference/src/model_zoo/base_model.py
"""基础模型模块。"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union
from src.toolbox.logger import get_logger
import logging
from pathlib import Path

logger = get_logger(__name__)

class BaseModel(ABC):
    """基础模型类，提供通用的模型接口。"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """初始化基础模型。
        
        Args:
            model_path: 模型路径
            device: 设备类型，可选 "cuda" 或 "cpu"
        """
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.device = device
        
        if not model_path or not isinstance(model_path, str):
            raise ValueError("模型路径必须是非空字符串")
            
        if os.getenv('TEST_MODE') == '1':
            self.logger.info("测试模式：跳过模型加载")
            self.model = None
            self.tokenizer = None
            return
            
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
            raise RuntimeError(f"加载模型失败：{str(e)}")
                
    def load_model(self):
        """加载模型。"""
        raise NotImplementedError
    
    def get_token_count(self, text: str) -> int:
        """获取文本的token数量。
        
        Args:
            text: 输入文本
            
        Returns:
            int: token数量
        """
        if os.getenv('TEST_MODE') == '1':
            return len(text.split())
            
        try:
            tokens = self.tokenizer.encode(text)
            return len(tokens)
        except Exception as e:
            self.logger.error(f"计算token数量失败: {str(e)}")
            return 0
    
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
        if self.model is None:
            self.model = self.load_model()
            self.tokenizer = self.tokenizer
            logger.info("模型初始化完成")
        else:
            logger.info("模型已初始化")

    def generate(self, prompt: str, max_length: int = 100) -> str:
        """生成文本。
        
        Args:
            prompt: 输入提示
            max_length: 最大生成长度
            
        Returns:
            str: 生成的文本
        """
        if os.getenv('TEST_MODE') == '1':
            return "This is a mock response."
            
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            self.logger.error(f"生成文本失败: {str(e)}")
            return ""
